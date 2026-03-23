// NeuroSan Studio – Azure Managed Application main deployment template
// Used for Azure Marketplace publishing (Managed Application offer type).
//
// Deploys:
//   - Azure Container Apps Environment (with Log Analytics)
//   - Backend Container App  (neuro-san grpc/http service)
//   - UI Container App       (web front-end)
//   - Managed Identity       (for ACR pull access)
//   - Key Vault              (for LLM API key storage)
//
// Parameters that map to the createUiDefinition.json outputs are marked [UI].

@description('[UI] Deployment location.')
param location string = resourceGroup().location

@description('[UI] Short name used as prefix for all resource names.')
@minLength(3)
@maxLength(16)
param appName string = 'neurosan'

@description('[UI] Container image tag for the backend (e.g. "latest" or a version like "1.2.3").')
param backendImageTag string = 'latest'

@description('[UI] Container image tag for the UI.')
param uiImageTag string = 'latest'

@description('[UI] Full ACR login server (e.g. myregistry.azurecr.io).')
param acrLoginServer string

@description('[UI] OpenAI API key – stored in Key Vault, injected via secret reference.')
@secure()
param openAiApiKey string = ''

@description('[UI] Anthropic API key – stored in Key Vault, injected via secret reference.')
@secure()
param anthropicApiKey string = ''

@description('[UI] Size/tier of Container App: "Consumption" (default, pay-per-use) or "Dedicated".')
@allowed(['Consumption', 'Dedicated'])
param containerAppTier string = 'Consumption'

// ── Derived names ────────────────────────────────────────────────────────────
var prefix = toLower(appName)
var logAnalyticsName = '${prefix}-logs'
var envName = '${prefix}-cae'
var kvName = '${prefix}-kv'
var backendAppName = '${prefix}-backend'
var uiAppName = '${prefix}-ui'
var identityName = '${prefix}-id'

var backendImage = '${acrLoginServer}/neuro-san-studio:${backendImageTag}'
var uiImage = '${acrLoginServer}/neuro-san-studio-ui:${uiImageTag}'

// ── Managed Identity (used for ACR pull + Key Vault read) ────────────────────
resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: identityName
  location: location
}

// ── Key Vault (stores LLM secrets) ───────────────────────────────────────────
resource kv 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: kvName
  location: location
  properties: {
    sku: { family: 'A', name: 'standard' }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true  // use RBAC rather than access policies
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    enabledForDeployment: false
    enabledForTemplateDeployment: false
  }
}

// Grant the managed identity permission to read secrets
resource kvSecretsUserRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(kv.id, identity.id, 'Key Vault Secrets User')
  scope: kv
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '4633458b-17de-408a-b874-0445c86b69e6' // Key Vault Secrets User
    )
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Store LLM keys as key vault secrets (only if provided)
resource openAiSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = if (!empty(openAiApiKey)) {
  parent: kv
  name: 'openai-api-key'
  properties: { value: openAiApiKey }
}

resource anthropicSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = if (!empty(anthropicApiKey)) {
  parent: kv
  name: 'anthropic-api-key'
  properties: { value: anthropicApiKey }
}

// ── ACR Pull role for the managed identity ────────────────────────────────────
// NOTE: The ACR must be in the same subscription.
// The deployer must grant AcrPull on their ACR to this identity.
// We output the identity principalId to make this post-deployment step easy.

// ── Log Analytics workspace ───────────────────────────────────────────────────
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
    features: { enableLogAccessUsingOnlyResourcePermissions: true }
  }
}

// ── Container Apps Environment ────────────────────────────────────────────────
resource cae 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: envName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
    workloadProfiles: containerAppTier == 'Consumption'
      ? []
      : [{ name: 'Consumption', workloadProfileType: 'Consumption' }]
  }
}

// ── Backend Container App ─────────────────────────────────────────────────────
resource backendApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: backendAppName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identity.id}': {} }
  }
  properties: {
    managedEnvironmentId: cae.id
    configuration: {
      ingress: {
        external: false          // backend is internal; the UI calls it
        targetPort: 8080
        transport: 'http'
        allowInsecure: false
      }
      registries: [
        {
          server: acrLoginServer
          identity: identity.id  // pull via managed identity (no stored password)
        }
      ]
      secrets: concat(
        empty(openAiApiKey) ? [] : [{
          name: 'openai-api-key'
          keyVaultUrl: openAiSecret.properties.secretUri
          identity: identity.id
        }],
        empty(anthropicApiKey) ? [] : [{
          name: 'anthropic-api-key'
          keyVaultUrl: anthropicSecret.properties.secretUri
          identity: identity.id
        }]
      )
    }
    template: {
      containers: [
        {
          name: 'neurosan-backend'
          image: backendImage
          resources: { cpu: json('1.0'), memory: '2Gi' }
          env: concat(
            empty(openAiApiKey) ? [] : [{ name: 'OPENAI_API_KEY', secretRef: 'openai-api-key' }],
            empty(anthropicApiKey) ? [] : [{ name: 'ANTHROPIC_API_KEY', secretRef: 'anthropic-api-key' }]
          )
          probes: [
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 8080, scheme: 'HTTP' }
              initialDelaySeconds: 20
              periodSeconds: 30
              failureThreshold: 3
            }
            {
              type: 'Readiness'
              httpGet: { path: '/health', port: 8080, scheme: 'HTTP' }
              initialDelaySeconds: 10
              periodSeconds: 10
            }
          ]
        }
      ]
      scale: { minReplicas: 1, maxReplicas: 5 }
    }
  }
}

// ── UI Container App ──────────────────────────────────────────────────────────
resource uiApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: uiAppName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identity.id}': {} }
  }
  properties: {
    managedEnvironmentId: cae.id
    configuration: {
      ingress: {
        external: true           // UI is internet-facing
        targetPort: 5001
        transport: 'http'
        allowInsecure: false
        corsPolicy: {
          allowedOrigins: ['*']
          allowedMethods: ['GET', 'POST', 'OPTIONS']
          allowedHeaders: ['*']
        }
      }
      registries: [
        {
          server: acrLoginServer
          identity: identity.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'neurosan-ui'
          image: uiImage
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: [
            {
              // Point the UI to the internal backend URL
              name: 'AGENT_API_URL'
              value: 'https://${backendApp.properties.configuration.ingress.fqdn}'
            }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: { path: '/', port: 5000, scheme: 'HTTP' }
              initialDelaySeconds: 15
              periodSeconds: 30
            }
          ]
        }
      ]
      scale: { minReplicas: 1, maxReplicas: 10 }
    }
  }
}

// ── Outputs ───────────────────────────────────────────────────────────────────
@description('Public URL of the NeuroSan UI.')
output uiUrl string = 'https://${uiApp.properties.configuration.ingress.fqdn}'

@description('Internal FQDN of the backend (for UI → backend communication).')
output backendFqdn string = backendApp.properties.configuration.ingress.fqdn

@description('Managed identity principal ID — use this to grant AcrPull on your ACR.')
output managedIdentityPrincipalId string = identity.properties.principalId

@description('Key Vault URI.')
output keyVaultUri string = kv.properties.vaultUri
